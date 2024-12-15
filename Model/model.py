from DialogueCRN import DialogueCRN
from MultiAttn import MultiAttnModel
from MLP import MLP
import torch
import torch.nn as nn
import math
from torch.nn.functional import gelu

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.actv = gelu
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.actv(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_count, model_dim, dropout=0.1):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_k = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_v = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_q = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None):
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                .view(batch_size, -1, head_count * dim_per_head)

        key = self.linear_k(key).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        query = self.linear_q(query).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e10)

        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)
        context = torch.matmul(drop_attn, value).transpose(1, 2). \
            contiguous().view(batch_size, -1, head_count * dim_per_head)
        output = self.linear(context)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, speaker_emb):
        L = x.size(1)
        pos_emb = self.pe[:, :L]


        if speaker_emb.size() == x.size():
            x = x
        else:
            x = x

        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, inputs_a, inputs_b, mask):
        if inputs_a.equal(inputs_b):
            if (iter != 0):
                inputs_b = self.layer_norm(inputs_b)
            else:
                inputs_b = inputs_b

            mask = mask.unsqueeze(1)
            context = self.self_attn(inputs_b, inputs_b, inputs_b, mask=mask)
        else:
            if (iter != 0):
                inputs_b = self.layer_norm(inputs_b)
            else:
                inputs_b = inputs_b

            mask = mask.unsqueeze(1)
            context = self.self_attn(inputs_a, inputs_a, inputs_b, mask=mask)

        if inputs_b.size() == context.size():
            out = self.dropout(context) + inputs_b
        else:

            inputs_b = inputs_b.permute(1, 0, 2)
            out = self.dropout(context) + inputs_b

        return self.feed_forward(out)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.layers = layers
        self.pos_emb = PositionalEncoding(d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_a, x_b, mask, speaker_emb):
        if x_a.equal(x_b):
            x_b = self.pos_emb(x_b, speaker_emb)
            x_b = self.dropout(x_b)
            for i in range(self.layers):
                x_b = self.transformer_inter[i](i, x_b, x_b, mask.eq(0))
        else:
            x_a = self.pos_emb(x_a, speaker_emb)
            x_a = self.dropout(x_a)
            x_b = self.pos_emb(x_b, speaker_emb)
            x_b = self.dropout(x_b)
            for i in range(self.layers):
                x_b = self.transformer_inter[i](i, x_a, x_b, mask.eq(0))
        return x_b

'''
MultiEMO consists of three key components: unimodal context modeling, multimodal fusion, and emotion classification. 
'''


class MultiEMO(nn.Module):

    def __init__(self, dataset, multi_attn_flag, roberta_dim, hidden_dim, dropout, num_layers,
                 model_dim, num_heads, D_m_audio, D_m_visual, D_g, D_p, D_e, D_h,
                 n_classes, n_speakers, listener_state, context_attention, D_a, dropout_rec, device):
        super().__init__()
        self.dataset = dataset
        self.multi_attn_flag = multi_attn_flag
        self.text_fc = nn.Linear(roberta_dim, model_dim)
        self.text_dialoguecrn = DialogueCRN(model_dim, n_classes, hidden_dim, num_layers, device)
        self.textf = nn.Conv1d(model_dim, hidden_dim, kernel_size=1, padding=0, bias=False)

        self.t_t = TransformerEncoder(d_model=hidden_dim, d_ff=model_dim, heads=num_heads, layers=2, dropout=dropout)
        self.a_t = TransformerEncoder(d_model=hidden_dim, d_ff=model_dim, heads=num_heads, layers=2, dropout=dropout)
        self.v_t = TransformerEncoder(d_model=hidden_dim, d_ff=model_dim, heads=num_heads, layers=2, dropout=dropout)

        self.audio_fc = nn.Linear(D_m_audio, model_dim)
        self.audio_dialoguecrn = DialogueCRN(model_dim, n_classes, hidden_dim, num_layers, device)
        self.acouf = nn.Conv1d(model_dim, hidden_dim, kernel_size=1, padding=0, bias=False)

        self.a_a = TransformerEncoder(d_model=hidden_dim, d_ff=model_dim, heads=num_heads, layers=2, dropout=dropout)
        self.t_a = TransformerEncoder(d_model=hidden_dim, d_ff=model_dim, heads=num_heads, layers=2, dropout=dropout)
        self.v_a = TransformerEncoder(d_model=hidden_dim, d_ff=model_dim, heads=num_heads, layers=2, dropout=dropout)

        self.visual_fc = nn.Linear(D_m_visual, model_dim)
        self.visual_dialoguecrn = DialogueCRN(model_dim, n_classes, hidden_dim, num_layers, device)
        self.visuf = nn.Conv1d(model_dim, hidden_dim, kernel_size=1, padding=0, bias=False)

        self.v_v = TransformerEncoder(d_model=hidden_dim, d_ff=model_dim, heads=num_heads, layers=2, dropout=dropout)
        self.t_v = TransformerEncoder(d_model=hidden_dim, d_ff=model_dim, heads=num_heads, layers=2, dropout=dropout)
        self.a_v = TransformerEncoder(d_model=hidden_dim, d_ff=model_dim, heads=num_heads, layers=2, dropout=dropout)

        self.multiattn = MultiAttnModel(num_layers, model_dim, num_heads, hidden_dim, dropout)
        self.fc = nn.Linear(model_dim * 3, model_dim)

        if self.dataset == 'MELD':
            self.mlp = MLP(model_dim, model_dim * 2, n_classes, dropout)
        elif self.dataset == 'IEMOCAP':
            self.mlp = MLP(model_dim, model_dim, n_classes, dropout)

    def forward(self, texts, audios, visuals, speaker_masks, utterance_masks, dia_len, padded_labels):
        spk_idx = torch.argmax(speaker_masks, -1)
        origin_spk_idx = spk_idx
        if self.n_speakers == 2:
            for i, x in enumerate(dia_len):
                spk_idx[i, x:] = (2 * torch.ones(origin_spk_idx[i].size(0) - x)).int().cuda()
        if self.n_speakers == 9:
            for i, x in enumerate(dia_len):
                spk_idx[i, x:] = (9 * torch.ones(origin_spk_idx[i].size(0) - x)).int().cuda()

        text_features = self.text_fc(texts.permute(1, 0, 2))
        audio_features = self.audio_fc(audios.permute(1, 0, 2))
        visual_features = self.visual_fc(visuals.permute(1, 0, 2))

        text_features = self.text_dialoguecrn(text_features, speaker_masks, utterance_masks).permute(1, 2, 0)
        audio_features = self.audio_dialoguecrn(audio_features, speaker_masks, utterance_masks).permute(1, 2, 0)
        visual_features = self.visual_dialoguecrn(visual_features,speaker_masks, utterance_masks).permute(1, 2, 0)

        text_features = self.textf(text_features).permute(2, 0, 1)
        audio_features = self.acouf(audio_features).permute(2, 0, 1)
        visual_features = self.visuf(visual_features).permute(2, 0, 1)

        t_t_transformer_out = self.t_t(text_features, text_features, utterance_masks, speaker_masks)
        a_t_transformer_out = self.a_t(audio_features, text_features, utterance_masks, speaker_masks)
        v_t_transformer_out = self.v_t(visual_features, text_features, utterance_masks, speaker_masks)

        a_a_transformer_out = self.a_a(audio_features, audio_features, utterance_masks, speaker_masks)
        t_a_transformer_out = self.t_a(text_features, audio_features, utterance_masks, speaker_masks)
        v_a_transformer_out = self.v_a(visual_features, audio_features, utterance_masks, speaker_masks)

        v_v_transformer_out = self.v_v(visual_features, visual_features, utterance_masks, speaker_masks)
        t_v_transformer_out = self.t_v(text_features, visual_features, utterance_masks, speaker_masks)
        a_v_transformer_out = self.a_v(audio_features, visual_features, utterance_masks, speaker_masks)

        Modality_Encoder_features1 = torch.cat((t_t_transformer_out, a_t_transformer_out, v_t_transformer_out), dim=-1)
        Modality_Encoder_features1 = Modality_Encoder_features1[:, :, :Modality_Encoder_features1.shape[2] // 3]
        Modality_Encoder_features2 = torch.cat((a_a_transformer_out, t_a_transformer_out, v_a_transformer_out), dim=-1)
        Modality_Encoder_features2 = Modality_Encoder_features2[:, :, :Modality_Encoder_features2.shape[2] // 3]
        Modality_Encoder_features3 = torch.cat((v_v_transformer_out, t_v_transformer_out, a_v_transformer_out), dim=-1)
        Modality_Encoder_features3 = Modality_Encoder_features3[:, :, :Modality_Encoder_features3.shape[2] // 3]

        Modality_Encoder_features1 = Modality_Encoder_features1.transpose(0, 1)
        Modality_Encoder_features2 = Modality_Encoder_features2.transpose(0, 1)
        Modality_Encoder_features3 = Modality_Encoder_features3.transpose(0, 1)

        if self.multi_attn_flag == True:
            fused_text_features, fused_audio_features, fused_visual_features = self.multiattn(
                Modality_Encoder_features1, Modality_Encoder_features2, Modality_Encoder_features3)
        else:
            fused_text_features, fused_audio_features, fused_visual_features = Modality_Encoder_features1, Modality_Encoder_features2, Modality_Encoder_features3

        fused_text_features = fused_text_features.reshape(-1, fused_text_features.shape[-1])
        fused_text_features = fused_text_features[padded_labels != -1]
        fused_audio_features = fused_audio_features.reshape(-1, fused_audio_features.shape[-1])
        fused_audio_features = fused_audio_features[padded_labels != -1]
        fused_visual_features = fused_visual_features.reshape(-1, fused_visual_features.shape[-1])
        fused_visual_features = fused_visual_features[padded_labels != -1]

        fused_features = torch.cat((fused_text_features, fused_audio_features, fused_visual_features), dim=-1)
        fc_outputs = self.fc(fused_features)
        mlp_outputs = self.mlp(fc_outputs)

        return fused_text_features, fused_audio_features, fused_visual_features, fc_outputs, mlp_outputs




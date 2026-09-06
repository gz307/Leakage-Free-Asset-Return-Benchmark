# -*- coding: utf-8 -*-

import os
import sys
import math

import torch
import torch.nn as nn

from types import (
    SimpleNamespace
)

from config import (
    OFFICIAL_DIR,
    SEQ_LEN
)


# ============================================================
# LSTM
# ============================================================

class LSTMModel(nn.Module):

    def __init__(
        self,
        hidden_size=32,
        num_layers=2,
        dropout=0.1
    ):

        super().__init__()

        self.lstm = nn.LSTM(

            input_size=1,

            hidden_size=hidden_size,

            num_layers=num_layers,

            batch_first=True,

            dropout=(
                dropout
                if num_layers > 1
                else 0
            )
        )

        self.head = nn.Linear(
            hidden_size,
            1
        )


    def forward(self, x):

        x, _ = self.lstm(x)

        return self.head(
            x[:, -1]
        )


# ============================================================
# Standard Transformer
# ============================================================

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        max_len=1000
    ):

        super().__init__()

        pe = torch.zeros(
            max_len,
            d_model
        )

        pos = torch.arange(
            max_len
        ).float().unsqueeze(1)

        div = torch.exp(

            torch.arange(
                0,
                d_model,
                2
            ).float()

            *
            (
                -math.log(10000.0)
                /
                d_model
            )
        )


        pe[:, 0::2] = torch.sin(
            pos * div
        )

        pe[:, 1::2] = torch.cos(
            pos * div
        )

        self.register_buffer(
            "pe",
            pe.unsqueeze(0)
        )


    def forward(self, x):

        return (
            x
            +
            self.pe[:, :x.size(1)]
        )


class TransformerModel(nn.Module):

    def __init__(
        self,
        d_model=32,
        nhead=4,
        num_layers=2,
        d_ff=128,
        dropout=0.1
    ):

        super().__init__()

        self.proj = nn.Linear(
            1,
            d_model
        )

        self.pos = PositionalEncoding(
            d_model
        )

        layer = nn.TransformerEncoderLayer(

            d_model=d_model,

            nhead=nhead,

            dim_feedforward=d_ff,

            dropout=dropout,

            activation="gelu",

            batch_first=True
        )

        self.encoder = (
            nn.TransformerEncoder(
                layer,
                num_layers=num_layers
            )
        )

        self.head = nn.Linear(
            d_model,
            1
        )


    def forward(self, x):

        x = self.proj(x)

        x = self.pos(x)

        x = self.encoder(x)

        return self.head(
            x[:, -1]
        )


# ============================================================
# Official Autoformer adapter
# ============================================================

class AutoformerAdapter(nn.Module):

    def __init__(
        self,
        official_model,
        label_len=30
    ):

        super().__init__()

        self.model = (
            official_model
        )

        self.label_len = (
            label_len
        )


    def forward(self, x):

        B, L, C = x.shape

        zeros = torch.zeros(
            B,
            1,
            C,
            device=x.device,
            dtype=x.dtype
        )


        x_dec = torch.cat(

            [
                x[
                    :,
                    -self.label_len:,
                    :
                ],

                zeros
            ],

            dim=1
        )


        mark_enc = torch.zeros(
            B,
            L,
            3,
            device=x.device,
            dtype=x.dtype
        )

        mark_dec = torch.zeros(
            B,
            self.label_len + 1,
            3,
            device=x.device,
            dtype=x.dtype
        )


        out = self.model(

            x,

            mark_enc,

            x_dec,

            mark_dec
        )


        if isinstance(
            out,
            tuple
        ):

            out = out[0]


        return out[
            :,
            -1,
            0
        ].reshape(
            -1,
            1
        )


# ============================================================
# Generic official wrappers
# ============================================================

class SimpleForecastAdapter(nn.Module):

    def __init__(
        self,
        model
    ):

        super().__init__()

        self.model = model


    def forward(self, x):

        out = self.model(x)

        if isinstance(
            out,
            tuple
        ):

            out = out[0]

        return out[
            :,
            -1,
            0
        ].reshape(
            -1,
            1
        )


class ITransformerAdapter(nn.Module):

    def __init__(
        self,
        model
    ):

        super().__init__()

        self.model = model


    def forward(self, x):

        out = self.model(
            x,
            None,
            None,
            None
        )


        if isinstance(
            out,
            tuple
        ):

            out = out[0]


        return out[
            :,
            -1,
            0
        ].reshape(
            -1,
            1
        )


# ============================================================
# Build model
#
# ============================================================

def build_model(
    model_name,
    params
):

    # ========================================================
    # LSTM
    # ========================================================

    if model_name == "LSTM":

        return LSTMModel(

            hidden_size=params[
                "hidden_size"
            ],

            num_layers=params[
                "num_layers"
            ],

            dropout=params[
                "dropout"
            ]
        )


    # ========================================================
    # Transformer
    # ========================================================

    if model_name == "Transformer":

        return TransformerModel(

            d_model=params[
                "d_model"
            ],

            nhead=params[
                "nhead"
            ],

            num_layers=params[
                "num_layers"
            ],

            d_ff=params[
                "d_ff"
            ],

            dropout=params[
                "dropout"
            ]
        )


    # ========================================================
    # Autoformer official
    # ========================================================

    if model_name == "Autoformer":

        root = os.path.join(
            OFFICIAL_DIR,
            "Autoformer"
        )

        sys.path.insert(
            0,
            root
        )

        from models.Autoformer import (
            Model
        )


        cfg = SimpleNamespace(

            seq_len=SEQ_LEN,

            label_len=30,

            pred_len=1,

            enc_in=1,

            dec_in=1,

            c_out=1,

            d_model=params[
                "d_model"
            ],

            n_heads=params[
                "nhead"
            ],

            e_layers=params[
                "num_layers"
            ],

            d_layers=1,

            d_ff=params[
                "d_ff"
            ],

            moving_avg=params[
                "moving_avg"
            ],

            factor=1,

            dropout=params[
                "dropout"
            ],

            embed="timeF",

            freq="d",

            activation="gelu",

            output_attention=False
        )


        official = Model(
            cfg
        )


        return AutoformerAdapter(
            official,
            label_len=30
        )


    # ========================================================
    # PatchTST official
    # ========================================================

    if model_name == "PatchTST":

        root = os.path.join(

            OFFICIAL_DIR,

            "PatchTST",

            "PatchTST_supervised"
        )

        sys.path.insert(
            0,
            root
        )


        from models.PatchTST import (
            Model
        )


        cfg = SimpleNamespace(

            seq_len=SEQ_LEN,

            pred_len=1,

            enc_in=1,

            patch_len=params[
                "patch_len"
            ],

            stride=params[
                "stride"
            ],

            padding_patch="end",

            d_model=params[
                "d_model"
            ],

            n_heads=params[
                "nhead"
            ],

            e_layers=params[
                "num_layers"
            ],

            d_ff=params[
                "d_ff"
            ],

            dropout=params[
                "dropout"
            ],

            fc_dropout=0.0,

            head_dropout=0.0,

            individual=False,

            revin=True,

            affine=True,

            subtract_last=False,

            decomposition=False,

            kernel_size=25,

            norm="BatchNorm",

            activation="gelu",

            res_attention=True,

            pre_norm=False,

            store_attn=False,

            pe="zeros",

            learn_pe=True
        )


        official = Model(
            cfg
        )


        return SimpleForecastAdapter(
            official
        )


    # ========================================================
    # iTransformer official
    # ========================================================

    if model_name == "iTransformer":

        root = os.path.join(
            OFFICIAL_DIR,
            "iTransformer"
        )

        sys.path.insert(
            0,
            root
        )


        from model.iTransformer import (
            Model
        )


        cfg = SimpleNamespace(

            seq_len=SEQ_LEN,

            pred_len=1,

            d_model=params[
                "d_model"
            ],

            n_heads=params[
                "nhead"
            ],

            e_layers=params[
                "num_layers"
            ],

            d_ff=params[
                "d_ff"
            ],

            factor=1,

            dropout=params[
                "dropout"
            ],

            activation="gelu",

            embed="timeF",

            freq="d",

            use_norm=True,

            output_attention=False,

            class_strategy="projection"
        )


        official = Model(
            cfg
        )


        return ITransformerAdapter(
            official
        )


    raise ValueError(
        model_name
    )

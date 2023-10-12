class UNetMidBlock2DSimpleCrossAttn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attention_head_dim=1,
        output_scale_factor=1.0,
        cross_attention_dim=1280,
        skip_time_act=False,
        only_cross_attention=False,
        cross_attention_norm=None,
    ):
        super().__init__()

        self.has_cross_attention = True

        self.attention_head_dim = attention_head_dim
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        self.num_heads = in_channels // self.attention_head_dim

        # there is always at least one resnet
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
                skip_time_act=skip_time_act,
            )
        ]
        attentions = []

        for _ in range(num_layers):
            processor = (
                AttnAddedKVProcessor2_0() if hasattr(F, "scaled_dot_product_attention") else AttnAddedKVProcessor()
            )

            attentions.append(
                Attention(
                    query_dim=in_channels,
                    cross_attention_dim=in_channels,
                    heads=self.num_heads,
                    dim_head=self.attention_head_dim,
                    added_kv_proj_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    bias=True,
                    upcast_softmax=True,
                    only_cross_attention=only_cross_attention,
                    cross_attention_norm=cross_attention_norm,
                    processor=processor,
                )
            )
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    skip_time_act=skip_time_act,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
    ):
        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
        lora_scale = cross_attention_kwargs.get("scale", 1.0)

        if attention_mask is None:
            # if encoder_hidden_states is defined: we are doing cross-attn, so we should use cross-attn mask.
            mask = None if encoder_hidden_states is None else encoder_attention_mask
        else:
            # when attention_mask is defined: we don't even check for encoder_attention_mask.
            # this is to maintain compatibility with UnCLIP, which uses 'attention_mask' param for cross-attn masks.
            # TODO: UnCLIP should express cross-attn mask via encoder_attention_mask param instead of via attention_mask.
            #       then we can simplify this whole if/else block to:
            #         mask = attention_mask if encoder_hidden_states is None else encoder_attention_mask
            mask = attention_mask

        hidden_states = self.resnets[0](hidden_states, temb, scale=lora_scale)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            # attn
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=mask,
                **cross_attention_kwargs,
            )

            # resnet
            hidden_states = resnet(hidden_states, temb, scale=lora_scale)

        return hidden_states

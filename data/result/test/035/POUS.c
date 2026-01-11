void LLM_WRAPPER_FB_init__(LLM_WRAPPER_FB *data__, BOOL retain) {
  __INIT_VAR(data__->EN,__BOOL_LITERAL(TRUE),retain)
  __INIT_VAR(data__->ENO,__BOOL_LITERAL(TRUE),retain)
  __INIT_VAR(data__->MS,0,retain)
  __INIT_VAR(data__->KMH,0,retain)
  __INIT_VAR(data__->KN,0,retain)
  __INIT_VAR(data__->MH,0,retain)
  __INIT_VAR(data__->YMS,0,retain)
  __INIT_VAR(data__->YKMH,0,retain)
  __INIT_VAR(data__->YKN,0,retain)
  __INIT_VAR(data__->YMH,0,retain)
}

// Code part
void LLM_WRAPPER_FB_body__(LLM_WRAPPER_FB *data__) {
  // Control execution
  if (!__GET_VAR(data__->EN)) {
    __SET_VAR(data__->,ENO,,__BOOL_LITERAL(FALSE));
    return;
  }
  else {
    __SET_VAR(data__->,ENO,,__BOOL_LITERAL(TRUE));
  }
  // Initialise TEMP variables

  __SET_VAR(data__->,YMS,,__GET_VAR(data__->MS,));
  __SET_VAR(data__->,YKMH,,(__GET_VAR(data__->KMH,) * 0.27777777777778));
  __SET_VAR(data__->,YKN,,(__GET_VAR(data__->KN,) * 0.5144444));
  __SET_VAR(data__->,YMH,,(__GET_VAR(data__->MH,) * 0.44704));

  goto __end;

__end:
  return;
} // LLM_WRAPPER_FB_body__() 





void AUTO_PROXY_PRG_init__(AUTO_PROXY_PRG *data__, BOOL retain) {
  LLM_WRAPPER_FB_init__(&data__->INSTANCE,retain);
}

// Code part
void AUTO_PROXY_PRG_body__(AUTO_PROXY_PRG *data__) {
  // Initialise TEMP variables

  LLM_WRAPPER_FB_body__(&data__->INSTANCE);

  goto __end;

__end:
  return;
} // AUTO_PROXY_PRG_body__() 






void LLM_WRAPPER_FB_init__(LLM_WRAPPER_FB *data__, BOOL retain) {
  __INIT_VAR(data__->EN,__BOOL_LITERAL(TRUE),retain)
  __INIT_VAR(data__->ENO,__BOOL_LITERAL(TRUE),retain)
  __INIT_VAR(data__->IN,0,retain)
  __INIT_VAR(data__->TO_LOWER,0,retain)
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

  if (((__GET_VAR(data__->IN,) > __BYTE_LITERAL(64)) && (__GET_VAR(data__->IN,) < __BYTE_LITERAL(91)))) {
    __SET_VAR(data__->,TO_LOWER,,(__GET_VAR(data__->IN,) | __BYTE_LITERAL(0x20)));
  } else if ((((__GET_VAR(data__->IN,) > __BYTE_LITERAL(191)) && (__GET_VAR(data__->IN,) < __BYTE_LITERAL(223))) && (__GET_VAR(data__->IN,) != __BYTE_LITERAL(215)))) {
    __SET_VAR(data__->,TO_LOWER,,(__GET_VAR(data__->IN,) | __BYTE_LITERAL(0x20)));
  } else {
    __SET_VAR(data__->,TO_LOWER,,__GET_VAR(data__->IN,));
  };

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






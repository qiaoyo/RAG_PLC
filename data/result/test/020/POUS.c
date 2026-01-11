void LLM_WRAPPER_FB_init__(LLM_WRAPPER_FB *data__, BOOL retain) {
  __INIT_VAR(data__->EN,__BOOL_LITERAL(TRUE),retain)
  __INIT_VAR(data__->ENO,__BOOL_LITERAL(TRUE),retain)
  __INIT_VAR(data__->IN,0,retain)
  __INIT_VAR(data__->VAL,0,retain)
  __INIT_VAR(data__->HYS,0,retain)
  __INIT_VAR(data__->Q,__BOOL_LITERAL(FALSE),retain)
  __INIT_VAR(data__->WIN,__BOOL_LITERAL(FALSE),retain)
  __INIT_VAR(data__->TMP,0,retain)
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

  __SET_VAR(data__->,TMP,,(__GET_VAR(data__->VAL,) - (__GET_VAR(data__->HYS,) * 0.5)));
  if ((__GET_VAR(data__->IN,) < __GET_VAR(data__->TMP,))) {
    __SET_VAR(data__->,Q,,__BOOL_LITERAL(FALSE));
    __SET_VAR(data__->,WIN,,__BOOL_LITERAL(FALSE));
  } else if ((__GET_VAR(data__->IN,) > (__GET_VAR(data__->TMP,) + __GET_VAR(data__->HYS,)))) {
    __SET_VAR(data__->,Q,,__BOOL_LITERAL(TRUE));
    __SET_VAR(data__->,WIN,,__BOOL_LITERAL(FALSE));
  } else {
    __SET_VAR(data__->,WIN,,__BOOL_LITERAL(TRUE));
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






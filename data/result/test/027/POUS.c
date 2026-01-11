void LLM_WRAPPER_FB_init__(LLM_WRAPPER_FB *data__, BOOL retain) {
  __INIT_VAR(data__->EN,__BOOL_LITERAL(TRUE),retain)
  __INIT_VAR(data__->ENO,__BOOL_LITERAL(TRUE),retain)
  __INIT_VAR(data__->T,0,retain)
  __INIT_VAR(data__->SDD_NH3,0,retain)
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

  if ((__GET_VAR(data__->T,) < -33.65)) {
    __SET_VAR(data__->,SDD_NH3,,EXP__REAL__REAL(
      (BOOL)__BOOL_LITERAL(TRUE),
      NULL,
      (REAL)(7.3396511649 - (1166.7498002 / (__GET_VAR(data__->T,) + 192.37)))));
  } else {
    __SET_VAR(data__->,SDD_NH3,,EXP__REAL__REAL(
      (BOOL)__BOOL_LITERAL(TRUE),
      NULL,
      (REAL)(11.210964456 - (2564.9140075 / (__GET_VAR(data__->T,) + 262.741)))));
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






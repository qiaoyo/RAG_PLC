void LLM_WRAPPER_FB_init__(LLM_WRAPPER_FB *data__, BOOL retain) {
  __INIT_VAR(data__->EN,__BOOL_LITERAL(TRUE),retain)
  __INIT_VAR(data__->ENO,__BOOL_LITERAL(TRUE),retain)
  __INIT_VAR(data__->DTI,0,retain)
  __INIT_VAR(data__->JD2000,0,retain)
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

  __SET_VAR(data__->,JD2000,,(UDINT_TO_REAL(
    (BOOL)__BOOL_LITERAL(TRUE),
    NULL,
    (UDINT)(__GET_VAR(data__->DTI,) - __UDINT_LITERAL(946728000))) / 86400.0));

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






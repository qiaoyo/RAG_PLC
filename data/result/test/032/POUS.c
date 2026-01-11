void LLM_WRAPPER_FB_init__(LLM_WRAPPER_FB *data__, BOOL retain) {
  __INIT_VAR(data__->EN,__BOOL_LITERAL(TRUE),retain)
  __INIT_VAR(data__->ENO,__BOOL_LITERAL(TRUE),retain)
  __INIT_VAR(data__->STR,__STRING_LITERAL(0,""),retain)
  __INIT_VAR(data__->CMP,__STRING_LITERAL(0,""),retain)
  __INIT_VAR(data__->IS_CC,__BOOL_LITERAL(FALSE),retain)
  __INIT_VAR(data__->L,0,retain)
  __INIT_VAR(data__->POS,0,retain)
  __INIT_VAR(data__->TEMP,__STRING_LITERAL(0,""),retain)
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

  __SET_VAR(data__->,L,,LEN__INT__STRING(
    (BOOL)__BOOL_LITERAL(TRUE),
    NULL,
    (STRING)__GET_VAR(data__->STR,)));
  if ((__GET_VAR(data__->L,) < LEN__INT__STRING(
    (BOOL)__BOOL_LITERAL(TRUE),
    NULL,
    (STRING)__GET_VAR(data__->CMP,)))) {
    __SET_VAR(data__->,IS_CC,,__BOOL_LITERAL(FALSE));
    goto __end;
  };
  __SET_VAR(data__->,TEMP,,MID__STRING__STRING__SINT__SINT(
    (BOOL)__BOOL_LITERAL(TRUE),
    NULL,
    (STRING)__GET_VAR(data__->STR,),
    (SINT)1,
    (SINT)LEN__SINT__STRING(
      (BOOL)__BOOL_LITERAL(TRUE),
      NULL,
      (STRING)__GET_VAR(data__->CMP,))));
  __SET_VAR(data__->,IS_CC,,EQ_STRING(__BOOL_LITERAL(TRUE), NULL, 2, __GET_VAR(data__->TEMP,), __GET_VAR(data__->CMP,)));

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






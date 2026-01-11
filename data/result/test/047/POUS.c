void LLM_WRAPPER_FB_init__(LLM_WRAPPER_FB *data__, BOOL retain) {
  __INIT_VAR(data__->EN,__BOOL_LITERAL(TRUE),retain)
  __INIT_VAR(data__->ENO,__BOOL_LITERAL(TRUE),retain)
  __INIT_VAR(data__->D0,__BOOL_LITERAL(FALSE),retain)
  __INIT_VAR(data__->D1,__BOOL_LITERAL(FALSE),retain)
  __INIT_VAR(data__->D2,__BOOL_LITERAL(FALSE),retain)
  __INIT_VAR(data__->D3,__BOOL_LITERAL(FALSE),retain)
  __INIT_VAR(data__->L,__BOOL_LITERAL(FALSE),retain)
  __INIT_VAR(data__->RST,__BOOL_LITERAL(FALSE),retain)
  __INIT_VAR(data__->Q0,__BOOL_LITERAL(FALSE),retain)
  __INIT_VAR(data__->Q1,__BOOL_LITERAL(FALSE),retain)
  __INIT_VAR(data__->Q2,__BOOL_LITERAL(FALSE),retain)
  __INIT_VAR(data__->Q3,__BOOL_LITERAL(FALSE),retain)
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

  if (__GET_VAR(data__->RST,)) {
    __SET_VAR(data__->,Q0,,__BOOL_LITERAL(FALSE));
    __SET_VAR(data__->,Q1,,__BOOL_LITERAL(FALSE));
    __SET_VAR(data__->,Q2,,__BOOL_LITERAL(FALSE));
    __SET_VAR(data__->,Q3,,__BOOL_LITERAL(FALSE));
  } else if (__GET_VAR(data__->L,)) {
    __SET_VAR(data__->,Q0,,__GET_VAR(data__->D0,));
    __SET_VAR(data__->,Q1,,__GET_VAR(data__->D1,));
    __SET_VAR(data__->,Q2,,__GET_VAR(data__->D2,));
    __SET_VAR(data__->,Q3,,__GET_VAR(data__->D3,));
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






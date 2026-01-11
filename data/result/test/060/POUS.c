void LLM_WRAPPER_FB_init__(LLM_WRAPPER_FB *data__, BOOL retain) {
  __INIT_VAR(data__->EN,__BOOL_LITERAL(TRUE),retain)
  __INIT_VAR(data__->ENO,__BOOL_LITERAL(TRUE),retain)
  __INIT_VAR(data__->I1,__BOOL_LITERAL(FALSE),retain)
  __INIT_VAR(data__->I2,__BOOL_LITERAL(FALSE),retain)
  __INIT_VAR(data__->TL,__time_to_timespec(1, 0, 0, 0, 0, 0),retain)
  __INIT_VAR(data__->Q1,__BOOL_LITERAL(FALSE),retain)
  __INIT_VAR(data__->Q2,__BOOL_LITERAL(FALSE),retain)
  TOF_init__(&data__->T1,retain);
  TOF_init__(&data__->T2,retain);
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

  __SET_VAR(data__->T1.,IN,,__GET_VAR(data__->I1,));
  __SET_VAR(data__->T1.,PT,,__GET_VAR(data__->TL,));
  TOF_body__(&data__->T1);
  __SET_VAR(data__->T2.,IN,,__GET_VAR(data__->I2,));
  __SET_VAR(data__->T2.,PT,,__GET_VAR(data__->TL,));
  TOF_body__(&data__->T2);
  __SET_VAR(data__->,Q1,,(__GET_VAR(data__->I1,) && !(__GET_VAR(data__->T2.Q,))));
  __SET_VAR(data__->,Q2,,(__GET_VAR(data__->I2,) && !(__GET_VAR(data__->T1.Q,))));

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






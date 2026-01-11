void LLM_WRAPPER_FB_init__(LLM_WRAPPER_FB *data__, BOOL retain) {
  __INIT_VAR(data__->EN,__BOOL_LITERAL(TRUE),retain)
  __INIT_VAR(data__->ENO,__BOOL_LITERAL(TRUE),retain)
  __INIT_VAR(data__->SET,__BOOL_LITERAL(FALSE),retain)
  __INIT_VAR(data__->IN,__BOOL_LITERAL(FALSE),retain)
  __INIT_VAR(data__->RST,__BOOL_LITERAL(FALSE),retain)
  __INIT_VAR(data__->TOGGLE_MODE,__BOOL_LITERAL(FALSE),retain)
  __INIT_VAR(data__->TIMEOUT,__time_to_timespec(1, 0, 0, 0, 0, 0),retain)
  __INIT_VAR(data__->Q,__BOOL_LITERAL(FALSE),retain)
  TON_init__(&data__->OFF,retain);
  __INIT_VAR(data__->EDGE,__BOOL_LITERAL(FALSE),retain)
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

  if (__GET_VAR(data__->OFF.Q,)) {
    __SET_VAR(data__->,Q,,__BOOL_LITERAL(FALSE));
  };
  if (__GET_VAR(data__->RST,)) {
    __SET_VAR(data__->,Q,,__BOOL_LITERAL(FALSE));
  } else if (__GET_VAR(data__->SET,)) {
    __SET_VAR(data__->,Q,,__BOOL_LITERAL(TRUE));
  } else if ((__GET_VAR(data__->IN,) && !(__GET_VAR(data__->EDGE,)))) {
    if (__GET_VAR(data__->TOGGLE_MODE,)) {
      __SET_VAR(data__->,Q,,!(__GET_VAR(data__->Q,)));
    } else {
      __SET_VAR(data__->,Q,,__BOOL_LITERAL(TRUE));
    };
  };
  __SET_VAR(data__->,EDGE,,__GET_VAR(data__->IN,));
  if (GT_TIME(__BOOL_LITERAL(TRUE), NULL, 2, __GET_VAR(data__->TIMEOUT,), __time_to_timespec(1, 0, 0, 0, 0, 0))) {
    __SET_VAR(data__->OFF.,IN,,__GET_VAR(data__->Q,));
    __SET_VAR(data__->OFF.,PT,,__GET_VAR(data__->TIMEOUT,));
    TON_body__(&data__->OFF);
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






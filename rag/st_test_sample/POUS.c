void LLM_WRAPPER_FB_init__(LLM_WRAPPER_FB *data__, BOOL retain) {
  __INIT_VAR(data__->EN,__BOOL_LITERAL(TRUE),retain)
  __INIT_VAR(data__->ENO,__BOOL_LITERAL(TRUE),retain)
  __INIT_VAR(data__->DIN,0,retain)
  __INIT_VAR(data__->E,__BOOL_LITERAL(TRUE),retain)
  __INIT_VAR(data__->RD,__BOOL_LITERAL(FALSE),retain)
  __INIT_VAR(data__->WD,__BOOL_LITERAL(FALSE),retain)
  __INIT_VAR(data__->RST,__BOOL_LITERAL(FALSE),retain)
  __INIT_VAR(data__->DOUT,0,retain)
  __INIT_VAR(data__->EMPTY,__BOOL_LITERAL(TRUE),retain)
  __INIT_VAR(data__->FULL,__BOOL_LITERAL(FALSE),retain)
  {
    static const __ARRAY_OF_DWORD_16 temp = {{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};
    __SET_VAR(data__->,FIFO,,temp);
  }
  __INIT_VAR(data__->PR,0,retain)
  __INIT_VAR(data__->PW,0,retain)
  __INIT_VAR(data__->COUNT,0,retain)
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
    __SET_VAR(data__->,PW,,0);
    __SET_VAR(data__->,PR,,0);
    __SET_VAR(data__->,COUNT,,0);
    __SET_VAR(data__->,FULL,,__BOOL_LITERAL(FALSE));
    __SET_VAR(data__->,EMPTY,,__BOOL_LITERAL(TRUE));
    __SET_VAR(data__->,DOUT,,__DWORD_LITERAL(0));
  } else if (__GET_VAR(data__->E,)) {
    if ((!(__GET_VAR(data__->EMPTY,)) && __GET_VAR(data__->RD,))) {
      __SET_VAR(data__->,DOUT,,__GET_VAR(data__->FIFO,.table[(__GET_VAR(data__->PR,)) - (0)]));
      __SET_VAR(data__->,PR,,((16 == 0)?0:((__GET_VAR(data__->PR,) + 1) % 16)));
      __SET_VAR(data__->,COUNT,,(__GET_VAR(data__->COUNT,) - 1));
      __SET_VAR(data__->,EMPTY,,(__GET_VAR(data__->COUNT,) == 0));
      __SET_VAR(data__->,FULL,,__BOOL_LITERAL(FALSE));
    };
    if ((!(__GET_VAR(data__->FULL,)) && __GET_VAR(data__->WD,))) {
      __SET_VAR(data__->,FIFO,.table[(__GET_VAR(data__->PW,)) - (0)],__GET_VAR(data__->DIN,));
      __SET_VAR(data__->,PW,,((16 == 0)?0:((__GET_VAR(data__->PW,) + 1) % 16)));
      __SET_VAR(data__->,COUNT,,(__GET_VAR(data__->COUNT,) + 1));
      __SET_VAR(data__->,FULL,,(__GET_VAR(data__->COUNT,) == 16));
      __SET_VAR(data__->,EMPTY,,__BOOL_LITERAL(FALSE));
    };
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






#ifndef __POUS_H
#define __POUS_H

#include "accessor.h"
#include "iec_std_lib.h"

// FUNCTION_BLOCK LLM_WRAPPER_FB
// Data part
typedef struct {
  // FB Interface - IN, OUT, IN_OUT variables
  __DECLARE_VAR(BOOL,EN)
  __DECLARE_VAR(BOOL,ENO)
  __DECLARE_VAR(BOOL,SET)
  __DECLARE_VAR(BOOL,D0)
  __DECLARE_VAR(BOOL,D1)
  __DECLARE_VAR(BOOL,D2)
  __DECLARE_VAR(BOOL,D3)
  __DECLARE_VAR(BOOL,D4)
  __DECLARE_VAR(BOOL,D5)
  __DECLARE_VAR(BOOL,D6)
  __DECLARE_VAR(BOOL,D7)
  __DECLARE_VAR(BOOL,CLR)
  __DECLARE_VAR(BOOL,RST)
  __DECLARE_VAR(BOOL,Q0)
  __DECLARE_VAR(BOOL,Q1)
  __DECLARE_VAR(BOOL,Q2)
  __DECLARE_VAR(BOOL,Q3)
  __DECLARE_VAR(BOOL,Q4)
  __DECLARE_VAR(BOOL,Q5)
  __DECLARE_VAR(BOOL,Q6)
  __DECLARE_VAR(BOOL,Q7)

  // FB private variables - TEMP, private and located variables
  __DECLARE_VAR(BOOL,EDGE)

} LLM_WRAPPER_FB;

void LLM_WRAPPER_FB_init__(LLM_WRAPPER_FB *data__, BOOL retain);
// Code part
void LLM_WRAPPER_FB_body__(LLM_WRAPPER_FB *data__);
// PROGRAM AUTO_PROXY_PRG
// Data part
typedef struct {
  // PROGRAM Interface - IN, OUT, IN_OUT variables

  // PROGRAM private variables - TEMP, private and located variables
  LLM_WRAPPER_FB INSTANCE;

} AUTO_PROXY_PRG;

void AUTO_PROXY_PRG_init__(AUTO_PROXY_PRG *data__, BOOL retain);
// Code part
void AUTO_PROXY_PRG_body__(AUTO_PROXY_PRG *data__);
#endif //__POUS_H

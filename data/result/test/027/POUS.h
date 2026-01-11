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
  __DECLARE_VAR(REAL,T)
  __DECLARE_VAR(REAL,SDD_NH3)

  // FB private variables - TEMP, private and located variables

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

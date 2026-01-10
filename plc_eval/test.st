
FUNCTION_BLOCK LLM_Wrapper_FB
VAR_INPUT
    DIN : DWORD;
    E : BOOL := TRUE;
    RD : BOOL;
    WD : BOOL;
    RST : BOOL;
END_VAR

VAR_OUTPUT
    DOUT : DWORD;
    EMPTY : BOOL := TRUE;
    FULL : BOOL;
END_VAR

VAR
    fifo : ARRAY[0..15] OF DWORD;
    pr : INT := 0;
    pw : INT := 0;
    count : INT := 0;
END_VAR

IF RST THEN
    pw := 0;
    pr := 0;
    count := 0;
    FULL := FALSE;
    EMPTY := TRUE;
    DOUT := DWORD#0;
ELSIF E THEN
    IF NOT EMPTY AND RD THEN
        DOUT := fifo[pr];
        pr := (pr + 1) MOD 16;
        count := count - 1;
        EMPTY := count = 0;
        FULL := FALSE;
    END_IF;
    IF NOT FULL AND WD THEN
        fifo[pw] := DIN;
        pw := (pw + 1) MOD 16;
        count := count + 1;
        FULL := count = 16;
        EMPTY := FALSE;
    END_IF;
END_IF;
END_FUNCTION_BLOCK




PROGRAM Auto_Proxy_Prg
  VAR
    instance : LLM_Wrapper_FB;
  END_VAR
  instance();
END_PROGRAM


CONFIGURATION Config0
  RESOURCE Res0 ON PLC
    TASK task0(INTERVAL := T#20ms, PRIORITY := 0);
    PROGRAM instance0 WITH task0 : Auto_Proxy_Prg;
  END_RESOURCE
END_CONFIGURATION

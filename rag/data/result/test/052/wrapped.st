VAR_IN_OUT
    PT : oscat_NW_BUF_LONG;
END_VAR

VAR_INPUT
    SIZE : UINT;
    START : UINT;
    STOP : UINT;
END_VAR

VAR_OUTPUT
    BUFFER_TO_STRING : oscat_STRING250;
END_VAR

VAR
    sta : UINT;
    stp : UINT;
    size2 : UINT;
    i : INT;
    BUF_TO_STRING : BUF_TO_STRING;
END_VAR

sta := MIN(START, SIZE - UINT#1);
stp := STOP;
(* 检查最大字符串长度 *)
IF stp - sta + UINT#1 > UINT#250 THEN
    stp := sta + UINT#250 - UINT#1;
END_IF;
stp := MIN(stp, SIZE - UINT#1);
size2 := stp - sta + UINT#1;
IF size2 > UINT#0 THEN
    FOR i := 0 TO 1 DO
        BUF_TO_STRING.REQ := i > 0;
        BUF_TO_STRING.BUF_FORMAT := TRUE;
        BUF_TO_STRING.BUF_OFFS := UINT_TO_DINT(START);
        BUF_TO_STRING.BUF_CNT := UINT_TO_DINT(size2);
        BUF_TO_STRING.BUFFER := pt;
        BUF_TO_STRING.DST := BUFFER_TO_STRING;
        BUF_TO_STRING();
        pt := BUF_TO_STRING.BUFFER;
        BUFFER_TO_STRING := BUF_TO_STRING.DST;
    END_FOR;
ELSE
    BUFFER_TO_STRING := '';
END_IF;
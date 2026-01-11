VAR_INPUT
    CB : BYTE;
    E : BOOL;
    PIN : STRING[8];
END_VAR

VAR_OUTPUT
    TP : BOOL;
END_VAR

VAR
    POS : INT := 1;
END_VAR

TP := FALSE;
IF E THEN
    IF CB = BYTE_TO_INT(GET_CHAR(PIN, POS)) THEN
        POS := POS + 1;
        IF POS > LEN(PIN) THEN
            (* 检测到正确的代码 *)
            TP := TRUE;
            POS := 1;
        END_IF;
    ELSE
        POS := 1;
    END_IF;
END_IF;
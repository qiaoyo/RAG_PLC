VAR_INPUT
    IN : oscat_STRING250;
    CX : STRING;
END_VAR

VAR_OUTPUT
    DEL_CHARS : oscat_STRING250;
END_VAR

VAR
    pos : INT;
    stop : INT;
    char : oscat_STRING1;
END_VAR

DEL_CHARS := IN;
stop := LEN(IN);
pos := 1;
WHILE pos <= stop DO
    char := MID(DEL_CHARS, pos, 1);
    IF FIND(CX, char) > 0 THEN
        DEL_CHARS := DELETE(DEL_CHARS, pos, 1);
        stop := stop - 1; (* 现在字符串长度减一 *)
    ELSE
        pos := pos + 1; (* 字符未找到，跳到下一个 *)
    END_IF;
END_WHILE;
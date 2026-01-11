VAR_INPUT
    IN : STRING;
END_VAR

VAR_OUTPUT
    STR_TO_DW : DWORD;
END_VAR

VAR
    i : INT;
    temp : DWORD;
    charValue : BYTE;
END_VAR

STR_TO_DW := 0;
temp := 0;

FOR i := 1 TO LEN(IN) DO
    charValue := BYTE(IN[i]);
    IF charValue >= BYTE('0') AND charValue <= BYTE('9') THEN
        temp := temp * 10 + (charValue - BYTE('0'));
    ELSIF charValue >= BYTE('A') AND charValue <= BYTE('F') THEN
        temp := temp * 16 + (charValue - BYTE('A') + 10);
    ELSIF charValue >= BYTE('a') AND charValue <= BYTE('f') THEN
        temp := temp * 16 + (charValue - BYTE('a') + 10);
    END_IF;
    STR_TO_DW := temp;
END_FOR;
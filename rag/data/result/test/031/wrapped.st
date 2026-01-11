VAR_INPUT
    C : BYTE;
END_VAR

VAR_OUTPUT
    CHARNAME : oscat_STRING10;
END_VAR

VAR
    CHARNAME2 : oscat_STRING10;
    pos : INT;
    i : INT;
    CHR_TO_STRING : CHR_TO_STRING;
    SETUP_CHARNAMES : SETUP_CHARNAMES;
    CHARNAMES : oscat_CHARNAMES;
END_VAR

SETUP_CHARNAMES(CHARNAMES := CHARNAMES);
CHARNAMES := SETUP_CHARNAMES.CHARNAMES;

IF C <> BYTE#0 THEN
    (* 构造由代码和 $ 符号组成的搜索字符串，并清空字符名称字符串 *)
    CHR_TO_STRING(C := C);
    CHARNAME := CHR_TO_STRING.CHR_TO_STRING;
    CHARNAME2 := CONCAT(CHARNAME, '&');
    CHARNAME := CONCAT(';', CHARNAME2);
    WHILE pos = 0 AND (i < 11) DO
        i := i + 1;
        pos := FIND(CHARNAMES[i], CHARNAME);
    END_WHILE;
    IF pos > 0 THEN
        CHARNAME := MID(CHARNAMES[i], 10, pos + 3);
        (* 搜索名称结束位置并截断 *)
        pos := FIND(CHARNAME, ';');
        CHARNAME := LEFT(CHARNAME, pos - 1);
    ELSE
        CHR_TO_STRING(C := C);
        CHARNAME := CHR_TO_STRING.CHR_TO_STRING;
    END_IF;
ELSE
    CHARNAME := '';
END_IF;
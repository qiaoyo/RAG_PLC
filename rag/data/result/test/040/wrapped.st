VAR_INPUT
    STR : oscat_STRING250;
    SRC : STRING;
    REP : STRING;
END_VAR

VAR_OUTPUT
    REPLACE_CHARS : oscat_STRING250;
END_VAR

VAR
    tmp : oscat_STRING250;
    a : INT;
    b : INT;
    c : oscat_STRING1;
    stp : INT;
END_VAR

REPLACE_CHARS := STR;
(* 确保 rep 和 src 长度相同且大于 0 *)
a := LEN(src);
b := LEN(rep);
IF a < b THEN
    rep := LEFT(rep, a);
ELSIF b < a THEN
    src := LEFT(src, b);
END_IF;

(* 查找字符串并在必要时替换 *)
stp := LEN(str);
FOR a := 1 TO stp DO
    c := MID(REPLACE_CHARS, 1, a);
    b := FIND(src, c);
    IF b > 0 THEN
        tmp := MID(rep, 1, b);
        REPLACE_CHARS := REPLACE(REPLACE_CHARS, tmp, 1, a);
    END_IF;
END_FOR;
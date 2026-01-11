VAR
    formatString : STRING;
    param1 : STRING;
    param2 : STRING;
    resultString : STRING;
    pos : INT;
    tempString : STRING;
END_VAR

(* Initialize the result string *)
resultString := formatString;

(* Find and replace placeholders with parameters *)
pos := FIND(resultString, '%s');

WHILE pos > 0 DO
    (* Replace the first occurrence of %s with param1 *)
    tempString := COPY(resultString, 1, pos - 1);
    tempString := CONCAT(tempString, param1);
    tempString := CONCAT(tempString, COPY(resultString, pos + 2, LEN(resultString)));
    resultString := tempString;

    (* Find the next occurrence of %s *)
    pos := FIND(resultString, '%s');
END_WHILE;

(* Repeat the process for other placeholders if needed *)
pos := FIND(resultString, '%d');
WHILE pos > 0 DO
    (* Replace the first occurrence of %d with param2 *)
    tempString := COPY(resultString, 1, pos - 1);
    tempString := CONCAT(tempString, param2);
    tempString := CONCAT(tempString, COPY(resultString, pos + 2, LEN(resultString)));
    resultString := tempString;

    (* Find the next occurrence of %d *)
    pos := FIND(resultString, '%d');
END_WHILE;
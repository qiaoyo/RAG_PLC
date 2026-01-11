VAR_INPUT
    SET : BOOL;
    IN : BOOL;
    RST : BOOL;
    TOGGLE_MODE : BOOL;
    TIMEOUT : TIME;
END_VAR

VAR_OUTPUT
    Q : BOOL;
END_VAR

VAR
    off : TON;
    edge : BOOL;
END_VAR

IF off.Q THEN
    Q := FALSE;
END_IF;

IF RST THEN
    Q := FALSE;
ELSIF SET THEN
    Q := TRUE;
ELSIF IN AND NOT edge THEN
    IF TOGGLE_MODE THEN
        Q := NOT Q;
    ELSE
        Q := TRUE;
    END_IF;
END_IF;

edge := IN;

IF TIMEOUT > T#0s THEN
    off(IN := Q, PT := TIMEOUT);
END_IF;
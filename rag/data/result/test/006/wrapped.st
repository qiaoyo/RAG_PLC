VAR_INPUT
    varName : STRING;
    value : DINT;
END_VAR

VAR
    varHandle : VAR_HANDLE;
    success : BOOL;
END_VAR

success := GET_VAR_HANDLE(varName, varHandle);

IF success THEN
    SET_VAR_VALUE(varHandle, value);
END_IF;
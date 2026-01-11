VAR_INPUT
	varName : STRING;
	value : DINT;
END_VAR

VAR
	varRef : REFERENCE TO DINT;
END_VAR

varRef := ADR(varName);
varRef^ := value;
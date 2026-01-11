VAR_INPUT
	A : oscat_vector_3;
END_VAR

VAR_OUTPUT
	V3_XANG : REAL;
END_VAR

VAR
	la : REAL;
END_VAR

la := SQRT(A.X * A.X + A.Y * A.Y + A.Z * A.Z);

IF la > 0.0 THEN
	V3_XANG := ACOS(A.X / la);
ELSE
	V3_XANG := 0.0;
END_IF;
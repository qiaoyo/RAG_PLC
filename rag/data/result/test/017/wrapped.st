VAR_INPUT
	X : oscat_complex;
	Y : oscat_complex;
END_VAR

VAR_OUTPUT
	CADD : oscat_complex;
END_VAR

CADD.re := X.re + Y.re;
CADD.im := X.im + Y.im;
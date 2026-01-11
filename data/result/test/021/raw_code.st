VAR_INPUT
    X : oscat_complex;
END_VAR

VAR_OUTPUT
    CEXP : oscat_complex;
END_VAR

VAR
    Temp : REAL;
END_VAR

Temp := EXP(X.re);
CEXP.re := Temp * COS(X.im);
CEXP.im := Temp * SIN(X.im);
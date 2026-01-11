VAR_INPUT
    X : REAL;
    T : REAL;
    U : REAL;
END_VAR

VAR
    tmp : REAL;
END_VAR

tmp := X - T;
CAUCHY := 0.318309886183791 * U / (U * U + tmp * tmp);
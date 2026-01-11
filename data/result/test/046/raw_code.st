VAR_INPUT
	ACT : REAL;
	SET : REAL;
	SUP : REAL;
	OFS : REAL;
	M_I : REAL;
	MAN : BOOL;
	RST : BOOL := FALSE;
	KP : REAL := 1.0;
	KI : REAL := 1.0;
	LL : REAL := -1000.0;
	LH : REAL := 1000.0;
END_VAR

VAR_OUTPUT
	Y : REAL;
	DIFF : REAL;
	LIM : BOOL;
END_VAR

VAR
	pi : FT_PIWL;
	co : CTRL_OUT;
END_VAR

DIFF := SET - ACT;
pi(in := DIFF, kp := KP, ki := KI, lim_l := LL, lim_h := LH, rst := RST);
co(ci := pi.Y, OFFSET := OFS, man_in := M_I, lim_l := LL, lim_h := LH, manual := MAN);
Y := co.Y;
LIM := co.LIM;
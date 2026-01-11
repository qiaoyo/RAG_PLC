VAR_INPUT
	IN : REAL;
	KP : REAL := 1.0;
	KI : REAL := 1.0;
	LIM_L : REAL := -1.0E38;
	LIM_H : REAL := 1.0E38;
	RST : BOOL := FALSE;
END_VAR

VAR_OUTPUT
	Y : REAL;
	LIM : BOOL;
END_VAR

VAR
	integ : FT_INT;
END_VAR

(*@KEY@: WORKSHEET
NAME: FT_PIW
IEC_LANGUAGE: ST
*)
(* 运行积分器 *)
integ(IN := IN, K := KI, RUN := NOT LIM, RST := RST);

(* 设置输出值 *)
Y := KP * IN + integ.Out;

(* 检查限制 *)
IF Y < LIM_L THEN
	Y := LIM_L;
	LIM := TRUE;
ELSIF Y > LIM_H THEN
	Y := LIM_H;
	LIM := TRUE;
ELSE
	LIM := FALSE;
END_IF;
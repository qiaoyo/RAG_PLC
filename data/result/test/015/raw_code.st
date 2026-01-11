VAR_INPUT
	IN0 : BOOL := TRUE;
	IN1 : BOOL := TRUE;
	IN2 : BOOL := TRUE;
	IN3 : BOOL := TRUE;
	START : BOOL;
	RST : BOOL;
	WAIT0 : TIME;
	DELAY0 : TIME;
	WAIT1 : TIME;
	DELAY1 : TIME;
	WAIT2 : TIME;
	DELAY2 : TIME;
	WAIT3 : TIME;
	DELAY3 : TIME;
	STOP_ON_ERROR : BOOL;
END_VAR

VAR_OUTPUT
	Q0 : BOOL;
	Q1 : BOOL;
	Q2 : BOOL;
	Q3 : BOOL;
	QX : BOOL;
	RUN : BOOL;
	STEP : INT := -1;
	STATUS : BYTE;
END_VAR

VAR
	last : TIME;
	edge : BOOL;
	tx : TIME;
	init : BOOL;
	T_PLC_MS : T_PLC_MS;
END_VAR

(* 读取 SPS 定时器 *)
T_PLC_MS();
	tx := UDINT_TO_TIME(T_PLC_MS.T_PLC_MS);

(* 启动时初始化 *)
IF NOT init THEN
	last := tx;
	init := TRUE;
	status := BYTE#110;
END_IF;

(* 异步复位 *)
IF rst THEN
	step := -1;
	Q0 := FALSE;
	Q1 := FALSE;
	Q2 := FALSE;
	Q3 := FALSE;
	status := BYTE#110;
	run := FALSE;

(* 启动输入的边沿重启顺序 *)
ELSIF start AND NOT edge THEN
	step := 0;
	last := tx;
	status := BYTE#111;
	Q0 := FALSE;
	Q1 := FALSE;
	Q2 := FALSE;
	Q3 := FALSE;
	run := TRUE;
END_IF;
edge := start;

(* 检查是否需要在状态上停止 *)
IF status > BYTE#0 AND status < BYTE#100 AND stop_on_error THEN RETURN; END_IF;

(* 顺序正在运行 *)
IF run AND step = 0 THEN
	IF NOT q0 AND in0 AND tx - last <= wait0 THEN
		Q0 := TRUE;
		last := tx;
	ELSIF NOT q0 AND tx - last > wait0 THEN
		status := BYTE#1;
		run := FALSE;
	ELSIF q0 AND tx - last >= delay0 THEN
		step := 1;
		last := tx;
	END_IF;
END_IF;
IF run AND step = 1 THEN
	IF NOT q1
VAR_INPUT
	T_UPPER : REAL;
	T_LOWER : REAL;
	PRESSURE : BOOL := TRUE;
	ENABLE : BOOL := TRUE;
	REQ_1 : BOOL;
	REQ_2 : BOOL;
	BOOST : BOOL;
	T_UPPER_MIN : REAL := REAL#50.0;
	T_UPPER_MAX : REAL := REAL#60.0;
	T_LOWER_ENABLE : BOOL;
	T_LOWER_MAX : REAL := REAL#60.0;
	T_REQUEST_1 : REAL := REAL#70.0;
	T_REQUEST_2 : REAL := REAL#50.0;
	T_REQUEST_HYS : REAL := REAL#5.0;
	T_PROTECT_HIGH : REAL := REAL#80.0;
	T_PROTECT_LOW : REAL := REAL#10.0;
END_VAR

VAR_OUTPUT
	HEAT : BOOL;
	ERROR : BOOL;
	STATUS : BYTE;
END_VAR

VAR
	edge : BOOL;
	boost_mode : BOOL;
	flag_0 : BOOL;
	flag_1 : BOOL;
	flag_2 : BOOL;
END_VAR

(* 读取传感器并检查数据有效性 *)
IF t_upper > t_protect_high THEN
	status := BYTE#1;
	heat := FALSE;
	error := TRUE;
ELSIF t_upper < t_protect_low THEN
	status := BYTE#2;
	heat := TRUE;
	error := TRUE;
ELSIF t_lower > T_protect_high AND t_lower_enable THEN
	status := BYTE#3;
	heat := FALSE;
	error := TRUE;
ELSIF t_lower < t_protect_low AND t_lower_enable THEN
	status := BYTE#4;
	heat := TRUE;
	error := TRUE;
ELSIF NOT pressure THEN
	status := BYTE#5;
	heat := FALSE;
	error := TRUE;
ELSIF req_1 OR req_2 OR enable OR boost THEN
	error := FALSE;

	(* 判断是否需要开启加热 *)
	IF boost AND NOT edge AND t_upper < t_upper_max THEN
		status := BYTE#101;
		heat := TRUE;
		boost_mode := TRUE;
	ELSIF enable AND t_upper < T_upper_min THEN
		status := BYTE#102;
		heat := TRUE;
	ELSIF req_1 AND t_upper < T_request_1 THEN
		status := BYTE#103;
		heat := TRUE;
	ELSIF req_2 AND t_upper < T_request_2 THEN
		status := BYTE#104;
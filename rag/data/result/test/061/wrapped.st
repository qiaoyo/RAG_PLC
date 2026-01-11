VAR_INPUT
	IN1 : BOOL;
	IN2 : BOOL;
	IN3 : BOOL;
	IN4 : BOOL;
	IN5 : BOOL;
	IN6 : BOOL;
	IN7 : BOOL;
	IN8 : BOOL;
END_VAR

VAR_OUTPUT
	OUT1 : BOOL;
	OUT2 : BOOL;
	OUT3 : BOOL;
	OUT4 : BOOL;
	OUT5 : BOOL;
	OUT6 : BOOL;
	OUT7 : BOOL;
	OUT8 : BOOL;
	ID : BYTE;
END_VAR

VAR_IN_OUT
	X : oscat_NET_VAR_DATA;
END_VAR

VAR
	init : BOOL;
	index : INT;
	index2 : INT;
	BYTE_TO_BITS : BYTE_TO_BITS;
END_VAR

IF NOT init THEN
	init := TRUE;
	X.ID_MAX := X.ID_MAX + USINT#1;
	ID := USINT_TO_BYTE(X.ID_MAX);
END_IF;

IF X.STATE > BYTE#0 AND X.ERROR_ID = BYTE#0 THEN
	index := X.index; (* PCWORX 工作区解决方案 *)
	IF INT_TO_UINT(index + 3) > X.BUF_SIZE THEN
		X.ERROR_ID := ID;
	ELSE
		IF X.STATE = BYTE#1 THEN (* 写入数据 *)
			X.S_BUF.BUFFER[index] := ID; (* ID *)
			index := index + 1;
			X.S_BUF.BUFFER[index] := BYTE#4; (* 类型 BOOL8 *)
			index := index + 1;
			X.S_BUF.BUFFER[index] := BYTE_OF_BIT(IN1,IN2,IN3,IN4,IN5,IN6,IN7,IN8);
			index := index + 1;
		ELSIF X.STATE = BYTE#2 THEN (* 读取数据 *)
			index2 := index + 1; (* PCWORX 工作区解决方案 *)
			IF X.R_BUF.BUFFER[index] <> ID OR X.R_BUF.BUFFER[index2] <> BYTE#4 THEN
				X.ERROR_ID := ID;
			ELSE
				index := index + 2;
				BYTE_TO_BITS(IN := X.R_BUF.BUFFER[index]);
				OUT1 := BYTE_TO_BITS.B0;
				OUT2 := BYTE_TO_BITS.B1;
				OUT3 := BYTE_TO_BITS.B2;
				OUT4 := BYTE_TO_BITS.B3;
				OUT5 := BYTE_TO_BITS.B4;
				OUT6 := BYTE_TO_BITS.B
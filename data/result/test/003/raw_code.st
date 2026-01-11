VAR_INPUT
	T : REAL;  // 温度 (°C)
	RN : REAL; // 参考电阻值 (Ω)
	B : REAL;  // 负温度系数 (1/°C)
END_VAR

VAR_OUTPUT
	RES_NTC : REAL;  // 计算得到的电阻值 (Ω)
END_VAR

RES_NTC := RN * EXP(B * (1.0 / (T + 273.15) - 1.0 / (25.0 + 273.15)));
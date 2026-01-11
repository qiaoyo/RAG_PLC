VAR_INPUT
    E : BOOL := TRUE;
    T0 : TIME;
    T1 : TIME;
    T2 : TIME;
    T3 : TIME;
    S0 : BOOL;
    SX : INT;
    SL : BOOL;
END_VAR

VAR_OUTPUT
    STATE : INT;
END_VAR

VAR
    tx : TIME;
    last : TIME;
    init : BOOL;
    T_PLC_MS : T_PLC_MS;
END_VAR

(* 读取系统时间 *)
T_PLC_MS();
tx := UDINT_TO_TIME(T_PLC_MS.T_PLC_MS);

(* 首次循环初始化 *)
IF NOT init THEN
    init := TRUE;
    last := tx;
END_IF;

IF E THEN
    IF SL THEN
        (* 当 SX > 0 时强制开始状态 SX *)
        state := LIMIT(0, SX, 3);
        last := tx;
        (* 这用于避免从调用程序重置 SX，虽然在 Codesys 中工作正常，但在其他系统中可能存在问题，因为我们写入输入 *)
        SL := FALSE;
    ELSE
        CASE state OF
            0 : (* 等待 T0 并切换到下一周期 *)
                IF tx - last >= T0 THEN
                    state := 1;
                    last := tx;
                END_IF;
            1 : (* 等待 T1 完成第一次周期 *)
                IF tx - last >= T1 THEN
                    state := 2;
                    last := tx;
                END_IF;
            2 : (* 等待 T2 完成第一次周期 *)
                IF tx - last >= T2 THEN
                    state := 3;
                    last := tx;
                END_IF;
            3 : (* 等待 T3 完成第二次周期 *)
                IF tx - last >= T3 THEN
                    IF S0 THEN state := 0; END_IF; (* 如果 S0 为假，则序列在状态 3 停止 *)
                    last := tx;
                END_IF;
        END_CASE;
    END_IF;
ELSE
    state := 0;
    last := tx;
END_IF;
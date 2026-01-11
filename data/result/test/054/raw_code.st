VAR_INPUT
    X : REAL;
    A_UP : REAL;
    A_DN : REAL;
    VU_MAX : REAL;
    VD_MAX : REAL;
    LIMIT_HIGH : REAL;
    LIMIT_LOW : REAL;
    RST : BOOL;
END_VAR

VAR_OUTPUT
    Y : REAL;
    V : REAL;
END_VAR

VAR
    cycle_time : TC_S;
    init : BOOL;
END_VAR

(* 读取系统时间（微秒） *)
cycle_time();

(* 确保输入值的范围 *)
A_up := MAX(0.0, A_UP);
A_dn := MIN(0.0, A_dn);
VU_max := MAX(0.0, VU_max);
VD_max := MIN(0.0, VD_MAX);

(* 计算输出偏移量 *)
IF rst OR NOT init THEN
    init := TRUE;
    Y := 0.0;
    V := 0.0;
ELSIF X = Y THEN
    V := 0.0;
ELSIF X > Y THEN
    (* 输出过低 >> 加速并在结束时刹车 *)
    (* 加速并限制在 vu_max *)
    V := MIN(V + A_UP * cycle_time.TC, VU_max);
    (* 计算能够刹车的最大速度并选择最小值 *)
    V := MIN(SQRT((Y - X) * 2.0 * A_DN), V);
    (* 计算输出并遵循限制 *)
    Y := LIMIT(LIMIT_LOW, Y + MIN(V * cycle_time.TC, X - Y), LIMIT_HIGH);
ELSIF X < Y THEN
    (* 输出过高 >> 降速并在结束时刹车 *)
    (* 加速并限制在 vd_max *)
    V := MAX(V + A_DN * cycle_time.TC, VD_max);
    (* 计算能够刹车的最大速度并选择最小值 *)
    V := MAX(-SQRT((Y - X) * 2.0 * A_UP), V);
    (* 计算输出并遵循限制 *)
    Y := LIMIT(LIMIT_LOW, Y + MAX(V * cycle_time.TC, X - Y), LIMIT_HIGH);
END_IF;
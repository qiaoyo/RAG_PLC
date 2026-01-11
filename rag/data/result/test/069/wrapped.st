VAR_INPUT
    IDATE : UDINT;
END_VAR

VAR_OUTPUT
    IS_LEAP_DAY : BOOL;
END_VAR

VAR
    YEAR : INT;
    DAY_OF_YEAR : INT;
END_VAR

(* Extract year and day of year from the timestamp *)
YEAR := INT((IDATE / UDINT#31557600) + 1970);
DAY_OF_YEAR := INT((IDATE MOD UDINT#31557600) / UDINT#86400) + 1;

(* Check if the year is a leap year *)
IS_LEAP_DAY := (YEAR MOD 4 = 0) AND (YEAR MOD 100 <> 0) OR (YEAR MOD 400 = 0);

(* Check if the day of year is February 29th *)
IF IS_LEAP_DAY THEN
    IS_LEAP_DAY := DAY_OF_YEAR = 60;
ELSE
    IS_LEAP_DAY := FALSE;
END_IF;
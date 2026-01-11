VAR_EXTERNAL
    LOG_CL : us_LOG_CONTROL;
END_VAR

(* Group: Variables *)
VAR
    FB_PRINT_SF : PRINT_SF;
    idx : INT;
END_VAR

(* Check if the new message level is higher than the current log level *)
IF DWORD_TO_BYTE(SHL(LOG_CL.NEW_MSG_OPTION, 16)) > LOG_CL.LEVEL THEN
    RETURN;
END_IF;

(* Set the maximum size of the log buffer *)
LOG_CL.SIZE := 400;

(* Format and print the new log message *)
FB_PRINT_SF.PRINTF_DATA := LOG_CL.PRINTF;
FB_PRINT_SF.STR := LOG_CL.NEW_MSG;
FB_PRINT_SF();
LOG_CL.NEW_MSG := FB_PRINT_SF.STR;
LOG_CL.PRINTF := FB_PRINT_SF.PRINTF_DATA;

(* Reset the log control structure if reset flag is set *)
IF LOG_CL.RESET THEN
    LOG_CL.RESET := FALSE;
    LOG_CL.RING_MODE := FALSE;
    LOG_CL.IDX := 0;
    LOG_CL.UPDATE_COUNT := LOG_CL.UPDATE_COUNT + UINT#1;
END_IF;

(* Manage the log messages in a circular buffer *)
IF LOG_CL.SIZE > 0 THEN
    IF LEN(LOG_CL.NEW_MSG) > 0 THEN
        (* Check if the buffer is full and switch to ring mode *)
        IF (LOG_CL.IDX >= LOG_CL.SIZE) THEN
            LOG_CL.RING_MODE := TRUE;
            LOG_CL.IDX := 0;
        END_IF;

        (* Update the log buffer with the new message *)
        LOG_CL.UPDATE_COUNT := LOG_CL.UPDATE_COUNT + UINT#1;
        LOG_CL.IDX := LOG_CL.IDX + INT#1;
        idx := LOG_CL.IDX;
        LOG_CL.MSG[idx] := LOG_CL.NEW_MSG;
        LOG_CL.MSG_OPTION[idx] := LOG_CL.NEW_MSG_OPTION;
        LOG_CL.NEW_MSG := '';
    END_IF;
END_IF;
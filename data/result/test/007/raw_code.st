VAR num j{5};
		VAR pose pose1:=[[0,0,0],[1,0,0,0]];
		j{1}:=joints{1};
		j{2}:=joints{2};
		j{3}:=joints{3};
		! j{4}:=joints{4};
		j{5}:=joints{4};
		FOR i FROM 1 TO 5 DO
			IF i=2 THEN
				j{3}:=j{3}-j{2};
			ENDIF
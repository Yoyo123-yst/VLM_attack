# Mode transitions

## h53:c07
- `refusal_margin`: r0:DENY → r1:INVALID → r2:INVALID → r3:INVALID → r4:INVALID
- `last_certificate`: r0:DENY → r1:INVALID → r2:DENY → r3:REFUSE → r4:INVALID
- `accumulated_certificate`: r0:DENY → r1:RELATED_SAFE → r2:INVALID → r3:RELATED_SAFE → r4:RELATED_SAFE
- `accumulated_exact_only`: r0:DENY → r1:DENY → r2:RELATED_SAFE → r3:INVALID → r4:RELATED_SAFE
- `targeted_prefix_earlystop`: r0:REFUSE → r1:RHC*
- `gateflip_fair`: r0:REFUSE → r1:REFUSE → r2:RELATED_SAFE → r3:DENY → r4:DENY

## h53:c08
- `refusal_margin`: r0:DENY → r1:DENY → r2:RHC*
- `last_certificate`: r0:DENY → r1:REFUSE → r2:INVALID → r3:RELATED_SAFE → r4:REFUSE
- `accumulated_certificate`: r0:DENY → r1:RELATED_SAFE → r2:RHC*
- `accumulated_exact_only`: r0:DENY → r1:RELATED_SAFE → r2:INVALID → r3:RELATED_SAFE → r4:INVALID
- `targeted_prefix_earlystop`: r0:DENY → r1:RHC*
- `gateflip_fair`: r0:REFUSE → r1:REFUSE → r2:REFUSE → r3:RELATED_SAFE → r4:REFUSE

## h64:c07
- `refusal_margin`: r0:RHC*
- `last_certificate`: r0:RHC*
- `accumulated_certificate`: r0:RHC*
- `accumulated_exact_only`: r0:INVALID → r1:RELATED_SAFE → r2:RELATED_SAFE → r3:RELATED_SAFE → r4:INVALID
- `targeted_prefix_earlystop`: r0:RELATED_SAFE → r1:RHC*
- `gateflip_fair`: r0:RHC*

## h64:c08
- `refusal_margin`: r0:RHC*
- `last_certificate`: r0:RHC*
- `accumulated_certificate`: r0:RHC*
- `accumulated_exact_only`: r0:RHC*
- `targeted_prefix_earlystop`: r0:DENY → r1:RELATED_SAFE → r2:RHC*
- `gateflip_fair`: r0:RHC*

## h66:c07
- `refusal_margin`: r0:RHC*
- `last_certificate`: r0:RELATED_SAFE → r1:RELATED_SAFE → r2:RHC*
- `accumulated_certificate`: r0:RELATED_SAFE → r1:RELATED_SAFE → r2:RELATED_SAFE → r3:RELATED_SAFE → r4:INVALID
- `accumulated_exact_only`: r0:RELATED_SAFE → r1:RELATED_SAFE → r2:INVALID → r3:RELATED_SAFE → r4:INVALID
- `targeted_prefix_earlystop`: r0:RHC*
- `gateflip_fair`: r0:RHC*

## h66:c08
- `refusal_margin`: r0:RHC*
- `last_certificate`: r0:RELATED_SAFE → r1:RELATED_SAFE → r2:RELATED_SAFE → r3:RELATED_SAFE → r4:REFUSE
- `accumulated_certificate`: r0:RELATED_SAFE → r1:RELATED_SAFE → r2:RELATED_SAFE → r3:RELATED_SAFE → r4:INVALID
- `accumulated_exact_only`: r0:RELATED_SAFE → r1:RELATED_SAFE → r2:REFUSE → r3:REFUSE → r4:RELATED_SAFE
- `targeted_prefix_earlystop`: r0:RHC*
- `gateflip_fair`: r0:RHC*

## h72:c07
- `refusal_margin`: r0:REFUSE → r1:RHC*
- `last_certificate`: r0:DENY → r1:REFUSE → r2:RELATED_SAFE → r3:INVALID → r4:DENY
- `accumulated_certificate`: r0:DENY → r1:INVALID → r2:RELATED_SAFE → r3:RELATED_SAFE → r4:RELATED_SAFE
- `accumulated_exact_only`: r0:DENY → r1:INVALID → r2:RELATED_SAFE → r3:RELATED_SAFE → r4:RELATED_SAFE
- `targeted_prefix_earlystop`: r0:REFUSE → r1:DENY → r2:DENY → r3:RHC*
- `gateflip_fair`: r0:REFUSE → r1:RHC*

## h72:c08
- `refusal_margin`: r0:RHC*
- `last_certificate`: r0:DENY → r1:REFUSE → r2:INVALID → r3:REFUSE → r4:DENY
- `accumulated_certificate`: r0:DENY → r1:RELATED_SAFE → r2:RHC*
- `accumulated_exact_only`: r0:DENY → r1:RELATED_SAFE → r2:INVALID → r3:INVALID → r4:INVALID
- `targeted_prefix_earlystop`: r0:INVALID → r1:RHC*
- `gateflip_fair`: r0:DENY → r1:RHC*

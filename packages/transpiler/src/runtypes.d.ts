import type { Call, CallSig, IProduction, Produced, produces, withCall } from "./runtypes-patch";
import { Category } from "./types";

declare module "runtypes/lib/runtype" {

  interface RuntypeBase<A = unknown> {
    [Call]?: CallSig;
    [Produced]?: IProduction<Category, "assert" | "control">;
    withCall: typeof withCall;
    produces: typeof produces;
  }
}

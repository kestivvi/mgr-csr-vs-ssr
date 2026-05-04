import { App } from "fresh";
import { app } from "./main.ts";

if (import.meta.main) {
  await app.listen({ port: 3000 });
}

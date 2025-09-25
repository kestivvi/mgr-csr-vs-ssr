import { define } from "../../utils.ts";

export default define.page(function Home(ctx) {
  return <h1>Hello, {ctx.params.name}</h1>;
});

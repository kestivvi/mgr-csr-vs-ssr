import { type RouteConfig, index, route } from "@react-router/dev/routes";

export default [
  index("routes/home.tsx"),
  route("dynamic/:name", "routes/dynamic/[name]/route.tsx"),
] satisfies RouteConfig;

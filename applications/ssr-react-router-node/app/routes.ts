import { type RouteConfig, index, route } from "@react-router/dev/routes";

export default [
  index("routes/home.tsx"),
  route("dynamic/:name", "routes/dynamic/[name]/route.tsx"),
  route("dynamic-app/:id", "routes/dynamic-app/[id]/route.tsx"),
] satisfies RouteConfig;

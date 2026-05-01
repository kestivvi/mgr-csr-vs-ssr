import { createFileRoute } from '@tanstack/solid-router'

export const Route = createFileRoute('/dynamic/$name')({
  component: DynamicComponent,
})

function DynamicComponent() {
  const params = Route.useParams()
  return (
    <div>
      <h1>Hello World</h1>
      <p>Dynamic ID: {params().name}</p>
    </div>
  )
}

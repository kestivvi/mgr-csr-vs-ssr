import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/dynamic/$name')({
  component: DynamicComponent,
})

function DynamicComponent() {
  const { name } = Route.useParams()
  return (
    <div>
      <h1>Hello World</h1>
      <p>Dynamic ID: {name}</p>
    </div>
  )
}

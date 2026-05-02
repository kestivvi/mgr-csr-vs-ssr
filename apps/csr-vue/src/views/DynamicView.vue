<template>
  <div>
    <h1>Hello World</h1>
    <p>Dynamic ID: {{ name }}</p>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'

const route = useRoute()
const name = computed(() => route.params.name as string)
</script>

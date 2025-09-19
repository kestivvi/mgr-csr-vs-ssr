<template>
  <h1>Hello, {{ name }}</h1>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'

const route = useRoute()
const name = computed(() => route.params.name as string)
</script>

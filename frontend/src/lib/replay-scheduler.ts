import { normalizeTimestampMs } from './time-utils'

interface ReplayEvent {
  type: 'message' | 'step' | 'trace' | 'dag' | 'ws_message'
  data: any
  timestamp: string
  originalIndex: number
  event_type?: string
}

export class ReplayScheduler {
  private events: ReplayEvent[] = []
  private currentIndex: number = 0
  private isPlaying: boolean = false
  private playbackSpeed: number = 1
  private startTime: number = 0
  private timeoutId: NodeJS.Timeout | null = null
  private onEvent: (event: ReplayEvent) => void
  private onComplete: () => void
  private skipUserMessageDelays: boolean = true
  private maxUserMessageDelay: number = 5000 // Maximum delay for user messages (5 seconds)

  constructor(onEvent: (event: ReplayEvent) => void, onComplete: () => void, skipUserMessageDelays: boolean = true) {
    this.onEvent = onEvent
    this.onComplete = onComplete
    this.skipUserMessageDelays = skipUserMessageDelays
  }

  setEvents(events: ReplayEvent[]): void {
    this.events = events.sort((a, b) => {
      // Handle both Unix timestamp (number) and ISO string formats
      return normalizeTimestampMs(a.timestamp) - normalizeTimestampMs(b.timestamp)
    })
    this.currentIndex = 0
  }

  setPlaybackSpeed(speed: number): void {
    this.playbackSpeed = speed
  }

  setSkipUserMessageDelays(skip: boolean): void {
    this.skipUserMessageDelays = skip
  }

  setMaxUserMessageDelay(delay: number): void {
    this.maxUserMessageDelay = delay
  }

  play(): void {
    if (this.events.length === 0) {
      this.onComplete()
      return
    }

    this.isPlaying = true
    this.startTime = Date.now()
    this.scheduleNextEvent()
  }

  pause(): void {
    this.isPlaying = false
    if (this.timeoutId) {
      clearTimeout(this.timeoutId)
      this.timeoutId = null
    }
  }

  stop(): void {
    this.isPlaying = false
    this.currentIndex = 0
    if (this.timeoutId) {
      clearTimeout(this.timeoutId)
      this.timeoutId = null
    }
  }

  private scheduleNextEvent(): void {
    if (!this.isPlaying || this.currentIndex >= this.events.length) {
      if (this.currentIndex >= this.events.length) {
        this.onComplete()
      }
      return
    }

    const currentEvent = this.events[this.currentIndex]
    let delay: number

    if (this.currentIndex === 0) {
      // First event plays immediately
      delay = 0
    } else {
      // Calculate delay based on time difference from previous event
      const prevEvent = this.events[this.currentIndex - 1]

      // Convert timestamps to milliseconds
      let currentTime: number
      let prevTime: number

      if (typeof currentEvent.timestamp === 'number') {
        currentTime = currentEvent.timestamp * 1000
      } else {
        currentTime = new Date(currentEvent.timestamp).getTime()
      }

      if (typeof prevEvent.timestamp === 'number') {
        prevTime = prevEvent.timestamp * 1000
      } else {
        prevTime = new Date(prevEvent.timestamp).getTime()
      }

      const actualDelay = currentTime - prevTime

      // Apply delay logic
      let adjustedDelay = actualDelay
      const isLastEvent = this.currentIndex === this.events.length - 1

      if (isLastEvent) {
        // Last event always plays immediately
        adjustedDelay = 0
      } else if (this.skipUserMessageDelays && actualDelay > this.maxUserMessageDelay) {
        const isCurrentUserMessage = this.isUserMessageEvent(currentEvent)
        const isPreviousUserMessage = this.currentIndex > 0 ? this.isUserMessageEvent(prevEvent) : false

        if (isCurrentUserMessage || isPreviousUserMessage) {
          // Skip long delays around user messages (continuation intervals)
          adjustedDelay = this.maxUserMessageDelay
        } else {
          // Keep all other delays as-is for realistic timing
        }
      }

      // Apply playback speed
      const originalDelay = adjustedDelay
      delay = Math.max(0, adjustedDelay / this.playbackSpeed)
    }

    // Schedule the event
    this.timeoutId = setTimeout(() => {
      this.onEvent(currentEvent)

      // If this was the last event, complete immediately
      if (this.currentIndex === this.events.length - 1) {
        this.currentIndex++ // Move to end position
        this.onComplete()
      } else {
        this.currentIndex++
        this.scheduleNextEvent()
      }
    }, delay)
  }

  private isUserMessageEvent(event: ReplayEvent): boolean {
    // Check if the event represents a user message
    if (event.type === 'trace' || event.type === 'ws_message') {
      const eventData = event.data as any
      // event_type can be in different locations
      const eventType = event.event_type || eventData.event_type
      return eventType === 'user_message'
    }
    return false
  }

  getProgress(): number {
    if (this.events.length === 0) return 0
    return Math.min(100, (this.currentIndex / this.events.length) * 100)
  }

  getCurrentEvent(): ReplayEvent | null {
    if (this.currentIndex >= this.events.length) return null
    return this.events[this.currentIndex]
  }

  getState(): {
    isPlaying: boolean
    currentIndex: number
    totalEvents: number
    progress: number
  } {
    return {
      isPlaying: this.isPlaying,
      currentIndex: this.currentIndex,
      totalEvents: this.events.length,
      progress: this.getProgress()
    }
  }
}

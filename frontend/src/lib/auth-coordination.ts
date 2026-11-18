export type AuthCoordinationUnavailableReason = "coordination_unavailable" | "operation_failed"

export type AuthCoordinationAttempt<T> =
  | { status: "completed"; value: T }
  | { status: "unavailable"; reason: AuthCoordinationUnavailableReason }

const DATABASE_NAME = "xagent-auth-coordination"
const DATABASE_VERSION = 1
const MUTATION_LOCK_STORE = "auth-mutations"
const REFRESH_LOCK_STORE = "auth-refreshes"
const KEEPALIVE_KEY = "active"

let databasePromise: Promise<IDBDatabase> | null = null

function getDatabase(): Promise<IDBDatabase> {
  if (databasePromise) return databasePromise
  databasePromise = new Promise((resolve, reject) => {
    if (typeof indexedDB === "undefined") {
      reject(new Error("IndexedDB is unavailable"))
      return
    }
    const request = indexedDB.open(DATABASE_NAME, DATABASE_VERSION)
    request.onupgradeneeded = () => {
      const database = request.result
      for (const store of [MUTATION_LOCK_STORE, REFRESH_LOCK_STORE]) {
        if (!database.objectStoreNames.contains(store)) database.createObjectStore(store)
      }
    }
    request.onsuccess = () => {
      const database = request.result
      database.onversionchange = () => {
        database.close()
        databasePromise = null
      }
      resolve(database)
    }
    request.onerror = () => {
      databasePromise = null
      reject(request.error ?? new Error("IndexedDB open failed"))
    }
    request.onblocked = () => {
      databasePromise = null
      reject(new Error("IndexedDB upgrade was blocked"))
    }
  })
  return databasePromise
}

function lockStore(name: string): string {
  return name.startsWith("xagent-auth-refresh:") ? REFRESH_LOCK_STORE : MUTATION_LOCK_STORE
}

async function withIndexedDbTransaction<T>(name: string, action: () => Promise<T>): Promise<AuthCoordinationAttempt<T>> {
  let database: IDBDatabase
  try {
    database = await getDatabase()
  } catch {
    return { status: "unavailable", reason: "coordination_unavailable" }
  }

  return new Promise(resolve => {
    let actionResult: AuthCoordinationAttempt<T> | null = null
    let actionStarted = false
    let keepAlive = true
    let settled = false
    let transactionEnded = false
    let transaction: IDBTransaction
    try {
      transaction = database.transaction(lockStore(name), "readwrite")
    } catch {
      resolve({ status: "unavailable", reason: "coordination_unavailable" })
      return
    }
    const store = transaction.objectStore(lockStore(name))

    const finish = (fallback: AuthCoordinationAttempt<T>) => {
      if (settled || !transactionEnded || (actionStarted && !actionResult)) return
      settled = true
      resolve(actionResult ?? fallback)
    }
    const pump = () => {
      if (!keepAlive) return
      const request = store.get(KEEPALIVE_KEY)
      request.onsuccess = pump
      request.onerror = () => {
        keepAlive = false
        try { transaction.abort() } catch { /* The transaction may already be inactive. */ }
      }
    }

    const acquired = store.put(name, KEEPALIVE_KEY)
    acquired.onsuccess = () => {
      actionStarted = true
      pump()
      void action().then(
        value => { actionResult = { status: "completed", value } },
        () => { actionResult = { status: "unavailable", reason: "operation_failed" } },
      ).finally(() => {
        keepAlive = false
        finish({ status: "unavailable", reason: "operation_failed" })
      })
    }
    acquired.onerror = () => {
      keepAlive = false
      try { transaction.abort() } catch { /* The transaction may already be inactive. */ }
    }
    transaction.oncomplete = () => {
      transactionEnded = true
      finish({ status: "unavailable", reason: "operation_failed" })
    }
    transaction.onabort = () => {
      transactionEnded = true
      finish({ status: "unavailable", reason: "operation_failed" })
    }
    transaction.onerror = () => { /* onabort reports the transaction failure */ }
  })
}

/** Serializes auth work across same-origin tabs, including HTTP deployments without Web Locks. */
export async function withAuthCoordination<T>(name: string, action: () => Promise<T>): Promise<AuthCoordinationAttempt<T>> {
  const locks = typeof navigator === "undefined" ? undefined : navigator.locks
  if (locks) {
    try {
      return { status: "completed", value: await locks.request(name, action) }
    } catch {
      return { status: "unavailable", reason: "operation_failed" }
    }
  }
  return withIndexedDbTransaction(name, action)
}

#pragma once
#ifndef THREADPOOL_H
#define THREADPOOL_H

#include <mutex>
#include <queue>
#include <functional>
#include <future>
#include <thread>
#include <utility>
#include <vector>
#include <iostream>

// Thread safe implementation of a Queue using a std::queue
template <typename T>
class SafeQueue
{
private:
    std::queue<T> m_queue; //����ģ�庯���������

    std::mutex m_mutex; // ���ʻ����ź���

public:
    SafeQueue() {}
    SafeQueue(SafeQueue &&other) {}
    ~SafeQueue() {}

    bool empty() // ���ض����Ƿ�Ϊ��
    {
        std::unique_lock<std::mutex> lock(m_mutex); // �����źű�����������ֹm_queue���ı�

        return m_queue.empty();
    }

    int size()
    {
        std::unique_lock<std::mutex> lock(m_mutex); // �����źű�����������ֹm_queue���ı�

        return m_queue.size();
    }

    // �������Ԫ��
    void enqueue(T &t)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_queue.emplace(t);
    }

    // ����ȡ��Ԫ��
    bool dequeue(T &t)
    {
        std::unique_lock<std::mutex> lock(m_mutex); // ���м���

        if (m_queue.empty())
            return false;
        t = std::move(m_queue.front()); // ȡ������Ԫ�أ����ض���Ԫ��ֵ����������ֵ����

        m_queue.pop(); // ������ӵĵ�һ��Ԫ��

        return true;
    }
};

class ThreadPool
{
private:
    class ThreadWorker // �����̹߳�����
    {
    private:
        int m_id; // ����id

        ThreadPool *m_pool; // �����̳߳�
    public:
        // ���캯��
        ThreadWorker(ThreadPool *pool, const int id) : m_pool(pool), m_id(id)
        {
        }

        // ����()����
        void operator()()
        {
            std::function<void()> func; // �������������func

            bool dequeued; // �Ƿ�����ȡ��������Ԫ��

            while (!m_pool->m_shutdown)
            {
                {
                    // Ϊ�̻߳��������������ʹ����̵߳����ߺͻ���
                    std::unique_lock<std::mutex> lock(m_pool->m_conditional_mutex);

                    // ����������Ϊ�գ�������ǰ�߳�
                    if (m_pool->m_queue.empty())
                    {
                        m_pool->m_conditional_lock.wait(lock); // �ȴ���������֪ͨ�������߳�
                    }

                    // ȡ����������е�Ԫ��
                    dequeued = m_pool->m_queue.dequeue(func);
                }

                // ����ɹ�ȡ����ִ�й�������
                if (dequeued) {
                    func();
                }
            }
        }
    };

    bool m_shutdown; // �̳߳��Ƿ�ر�

    SafeQueue<std::function<void()>> m_queue; // ִ�к�����ȫ���У����������

    std::vector<std::thread> m_threads; // �����̶߳���

    std::mutex m_conditional_mutex; // �߳��������������

    std::condition_variable m_conditional_lock; // �̻߳��������������̴߳������߻��߻���״̬

public:
    // �̳߳ع��캯��
    ThreadPool(const int n_threads = 4)
        : m_threads(std::vector<std::thread>(n_threads)), m_shutdown(false)
    {
    }
    ThreadPool(const ThreadPool &) = delete;

    ThreadPool(ThreadPool &&) = delete;

    ThreadPool &operator=(const ThreadPool &) = delete;

    ThreadPool &operator=(ThreadPool &&) = delete;

    // Inits thread pool
    void init()
    {
        for (int i = 0; i < m_threads.size(); ++i)
        {
            m_threads.at(i) = std::thread(ThreadWorker(this, i)); // ���乤���߳�
        }
    }

    // Waits until threads(those who has a task) finish their current task and shutdowns the pool
    void shutdown()
    {
        m_shutdown = true;
        m_conditional_lock.notify_all(); // ֪ͨ���������й����߳�

        for (int i = 0; i < m_threads.size(); ++i)
        {
            if (m_threads.at(i).joinable()) // �ж��߳��Ƿ��ڵȴ�
            {
                m_threads.at(i).join(); // ���̼߳��뵽�ȴ�����
            }
        }
    }

    // Waits until all the tasks finished by threads and shutdowns the pool
    void myShutdown()
    {
        while(!m_queue.empty()) {

        }
        shutdown();
    }
    // Submit a function to be executed asynchronously by the pool
    template <typename F, typename... Args>
    auto submit(F &&f, Args &&...args) -> std::future<decltype(f(args...))>
    {
        // Create a function with bounded parameter ready to execute
        std::function<decltype(f(args...))()> func = std::bind(std::forward<F>(f), std::forward<Args>(args)...); // ���Ӻ����Ͳ������壬���⺯�����ͣ���������ֵ����

        // Encapsulate it into a shared pointer in order to be able to copy construct
        auto task_ptr = std::make_shared<std::packaged_task<decltype(f(args...))()>>(func);

        // Warp packaged task into void function
        std::function<void()> warpper_func = [task_ptr]()
        {
            (*task_ptr)();
        };

        // ����ͨ�ð�ȫ�����������ѹ�밲ȫ����
        m_queue.enqueue(warpper_func);

        // ����һ���ȴ��е��߳�
        m_conditional_lock.notify_one();

        // ������ǰע�������ָ��
        return task_ptr->get_future();
    }
};

#endif // THREADPOOL_H

